# prediction_function.py
# À placer à la racine de votre repo

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os
from datetime import datetime

def predict_electricity_price(date, temperature, precipitation, snow_depth, sunshine_hours, 
                             historical_prices=None):
    """
    Prédit le prix de l'électricité en fonction des variables météorologiques.
    
    Args:
        date: Date pour laquelle la prédiction est faite (datetime)
        temperature: Température moyenne (°C)
        precipitation: Précipitations (mm)
        snow_depth: Épaisseur de neige (cm)
        sunshine_hours: Durée d'ensoleillement (heures)
        historical_prices: Liste optionnelle des 7 derniers prix (pour les variables retardées)
        
    Returns:
        float: Prix prédit de l'électricité (EUR/MWh)
    """
    # Vérifier si les modèles existent
    model_files = os.listdir('./models')
    model1_file = [f for f in model_files if f.startswith('model1_') and f.endswith('.pkl')]
    model2_file = [f for f in model_files if f.startswith('model2_') and f.endswith('.pkl')]
    
    if not model1_file or not model2_file:
        raise ValueError("Les modèles n'ont pas été trouvés. Veuillez d'abord entraîner les modèles.")
    
    # Charger les modèles
    model1 = joblib.load(f'./models/{model1_file[0]}')
    model2 = joblib.load(f'./models/{model2_file[0]}')
    
    # Essayer de charger le modèle MLP d'abord, puis LSTM si le MLP n'est pas disponible
    if os.path.exists('./models/model3_MLP.h5'):
        model3 = load_model('./models/model3_MLP.h5')
        use_lstm = False
    elif os.path.exists('./models/model3_LSTM.h5'):
        model3 = load_model('./models/model3_LSTM.h5')
        use_lstm = True
    else:
        raise ValueError("Aucun modèle de réseau de neurones trouvé.")
    
    # Charger le scaler
    scaler = joblib.load('./models/scaler.pkl')
    
    # Créer les caractéristiques
    features = {
        'Temperature': temperature,
        'Precipitation': precipitation,
        'SnowDepth': snow_depth,
        'SunshineHours': sunshine_hours,
        'Month': date.month,
        'DayOfWeek': date.weekday(),
        'Season': date.month % 12 // 3 + 1  # 1: Hiver, 2: Printemps, 3: Été, 4: Automne
    }
    
    # Ajouter les variables retardées si disponibles
    if historical_prices is not None and len(historical_prices) >= 7:
        for i in range(1, 8):
            features[f'Price_lag_{i}'] = historical_prices[-i]
        
        # Ajouter la moyenne mobile sur 7 jours
        features['Price_MA7'] = sum(historical_prices[-7:]) / 7
        
        # Température MA7 (non disponible dans ce cas simple, nous utilisons la valeur actuelle)
        features['Temp_MA7'] = temperature
    else:
        # Si pas d'historique, utiliser des valeurs par défaut ou extraire des données d'entraînement
        try:
            # Charger les données fusionnées si disponibles
            merged_data = pd.read_csv('./data/merged_data.csv')
            # Utiliser les dernières valeurs comme approximation
            for i in range(1, 8):
                features[f'Price_lag_{i}'] = merged_data['Price'].iloc[-i]
            features['Price_MA7'] = merged_data['Price'].iloc[-7:].mean()
            features['Temp_MA7'] = merged_data['Temperature'].iloc[-7:].mean()
        except:
            # Si pas de données disponibles, utiliser des valeurs par défaut
            for i in range(1, 8):
                features[f'Price_lag_{i}'] = 50.0  # Valeur par défaut
            features['Price_MA7'] = 50.0
            features['Temp_MA7'] = temperature
    
    # Convertir en array numpy
    X = pd.DataFrame([features])
    
    # S'assurer que les colonnes sont dans le même ordre que lors de l'entraînement
    with open('./data/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Réorganiser les colonnes
    X = X.reindex(columns=feature_names)
    
    # Normaliser les données
    X_scaled = scaler.transform(X)
    
    # Faire les prédictions
    pred1 = model1.predict(X_scaled)[0]
    pred2 = model2.predict(X_scaled)[0]
    
    if use_lstm:
        # Pour LSTM, nous avons besoin d'une séquence
        # C'est simplifié ici, dans un cas réel il faudrait construire correctement la séquence
        X_lstm = np.reshape(X_scaled, (1, 1, X_scaled.shape[1]))
        pred3 = model3.predict(X_lstm)[0][0]
    else:
        pred3 = model3.predict(X_scaled)[0][0]
    
    # Moyenne des 3 modèles
    ensemble_pred = (pred1 + pred2 + pred3) / 3
    
    return {
        'Linear Regression': float(pred1),
        'Ensemble Method': float(pred2),
        'Neural Network': float(pred3),
        'Ensemble (Average)': float(ensemble_pred)
    }

# Exemple d'utilisation
if __name__ == "__main__":
    # Date actuelle
    now = datetime.now()
    
    # Exemple de données météo
    temperature = 15.0  # °C
    precipitation = 2.5  # mm
    snow_depth = 0.0    # cm
    sunshine_hours = 8.0 # heures
    
    # Exemple d'historique des prix (7 derniers jours)
    historical_prices = [45.2, 47.8, 50.1, 49.3, 48.7, 51.2, 52.5]
    
    # Prédiction
    try:
        prediction = predict_electricity_price(now, temperature, precipitation, snow_depth, 
                                              sunshine_hours, historical_prices)
        print("Prédiction des prix de l'électricité (EUR/MWh):")
        for model, price in prediction.items():
            print(f"{model}: {price:.2f}")
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        print("Veuillez d'abord exécuter les scripts d'entraînement des modèles.")