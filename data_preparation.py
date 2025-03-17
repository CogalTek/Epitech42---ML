# data_preparation.py
# À placer à la racine de votre repo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Charger les données de prix d'électricité
def load_electricity_data():
    # Choisir les fichiers de la zone SE3 (puisque les données météo concernent SE3)
    electricity_files = [f for f in os.listdir('./habo_plast/electricity') if f.startswith('SE3_') and f.endswith('entsoe.csv')]
    
    print(f"Fichiers trouvés: {electricity_files}")
    
    dfs = []
    for file in electricity_files:
        try:
            df = pd.read_csv(f'./habo_plast/electricity/{file}')
            # Examiner la structure du fichier
            if dfs == []:  # seulement pour le premier fichier
                print(f"Premières lignes de {file}:")
                print(df.head())
                print(f"Types de données: {df.dtypes}")
            
            # S'assurer que la colonne de prix est numérique
            if 'Day-ahead Price [EUR/MWh]' in df.columns:
                # Vérifier si la colonne contient des valeurs non numériques
                try:
                    df['Day-ahead Price [EUR/MWh]'] = pd.to_numeric(df['Day-ahead Price [EUR/MWh]'], errors='coerce')
                    print(f"Conversion numérique réussie pour {file}")
                except Exception as e:
                    print(f"Erreur lors de la conversion en numérique: {e}")
                    # Afficher quelques valeurs problématiques
                    print("Exemples de valeurs dans la colonne prix:")
                    print(df['Day-ahead Price [EUR/MWh]'].iloc[:5].tolist())
            
            dfs.append(df)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier {file}: {e}")
    
    # Concaténer tous les fichiers
    electricity_df = pd.concat(dfs, ignore_index=True)
    
    # Convertir les dates
    try:
        # Si le format est "DD.MM.YYYY HH:MM - DD.MM.YYYY HH:MM"
        electricity_df['Date'] = electricity_df['MTU (CET/CEST)'].apply(
            lambda x: pd.to_datetime(x.split(' - ')[0], format='%d.%m.%Y %H:%M')
        )
    except Exception as e:
        print(f"Erreur lors de la conversion des dates: {e}")
        return pd.DataFrame()  # Retourner un DataFrame vide en cas d'erreur
    
    # Agréger par jour pour correspondre aux données météo quotidiennes
    electricity_df['Day'] = electricity_df['Date'].dt.date
    
    # S'assurer que la colonne de prix est numérique avant l'agrégation
    print("Vérification finale des types de données avant l'agrégation:")
    print(electricity_df.dtypes)
    
    # Utiliser une méthode plus robuste pour l'agrégation
    try:
        # Filtrer les valeurs non numériques
        electricity_df = electricity_df[pd.to_numeric(electricity_df['Day-ahead Price [EUR/MWh]'], errors='coerce').notna()]
        electricity_df['Day-ahead Price [EUR/MWh]'] = pd.to_numeric(electricity_df['Day-ahead Price [EUR/MWh]'])
        
        # Grouper et calculer la moyenne
        daily_electricity = electricity_df.groupby('Day')['Day-ahead Price [EUR/MWh]'].mean().reset_index()
        daily_electricity.rename(columns={'Day-ahead Price [EUR/MWh]': 'Price'}, inplace=True)
    except Exception as e:
        print(f"Erreur lors de l'agrégation: {e}")
        
        # Alternative: traiter manuellement chaque groupe
        unique_days = electricity_df['Day'].unique()
        prices = []
        
        for day in unique_days:
            day_data = electricity_df[electricity_df['Day'] == day]
            try:
                # Convertir en numérique et ignorer les valeurs non convertibles
                numeric_prices = pd.to_numeric(day_data['Day-ahead Price [EUR/MWh]'], errors='coerce')
                avg_price = numeric_prices.mean()
                prices.append({'Day': day, 'Price': avg_price})
            except Exception as e_inner:
                print(f"Erreur pour le jour {day}: {e_inner}")
        
        daily_electricity = pd.DataFrame(prices)
    
    print("Données de prix quotidiens traitées avec succès.")
    return daily_electricity

# Charger les données météorologiques
def load_weather_data():
    # Température (paramètre 2)
    temp_files = [f for f in os.listdir('./habo_plast/smhi_data_2022-today/parameter_2') if f.endswith('.csv')]
    temp_dfs = []
    
    for file in temp_files:
        try:
            df = pd.read_csv(f'./habo_plast/smhi_data_2022-today/parameter_2/{file}')
            df['station'] = file.split('-')[0].replace('station_', '')
            temp_dfs.append(df)
        except:
            print(f"Problème avec le fichier température: {file}")
    
    temp_df = pd.concat(temp_dfs, ignore_index=True)
    temp_df['Date'] = pd.to_datetime(temp_df['Datum'])
    
    # Calculer la température moyenne par jour sur toutes les stations
    temp_daily = temp_df.groupby('Date')['Lufttemperatur'].mean().reset_index()
    temp_daily.rename(columns={'Lufttemperatur': 'Temperature'}, inplace=True)
    
    # Précipitations (paramètre 5)
    precip_files = [f for f in os.listdir('./habo_plast/smhi_data_2022-today/parameter_5') if f.endswith('.csv')]
    precip_dfs = []
    
    for file in precip_files:
        try:
            df = pd.read_csv(f'./habo_plast/smhi_data_2022-today/parameter_5/{file}')
            df['station'] = file.split('-')[0].replace('station_', '')
            precip_dfs.append(df)
        except:
            print(f"Problème avec le fichier précipitation: {file}")
    
    precip_df = pd.concat(precip_dfs, ignore_index=True)
    precip_df['Date'] = pd.to_datetime(precip_df['Datum'])
    
    # Calculer les précipitations moyennes par jour sur toutes les stations
    precip_daily = precip_df.groupby('Date')['Nederbördsmängd'].mean().reset_index()
    precip_daily.rename(columns={'Nederbördsmängd': 'Precipitation'}, inplace=True)
    
    # Épaisseur de neige (paramètre 8)
    snow_files = [f for f in os.listdir('./habo_plast/smhi_data_2022-today/parameter_8') if f.endswith('.csv')]
    snow_dfs = []
    
    for file in snow_files:
        try:
            df = pd.read_csv(f'./habo_plast/smhi_data_2022-today/parameter_8/{file}')
            df['station'] = file.split('-')[0].replace('station_', '')
            snow_dfs.append(df)
        except:
            print(f"Problème avec le fichier neige: {file}")
    
    snow_df = pd.concat(snow_dfs, ignore_index=True)
    snow_df['Date'] = pd.to_datetime(snow_df['Datum'])
    
    # Calculer l'épaisseur de neige moyenne par jour sur toutes les stations
    snow_daily = snow_df.groupby('Date')['Snödjup'].mean().reset_index()
    snow_daily.rename(columns={'Snödjup': 'SnowDepth'}, inplace=True)
    
    # Durée d'ensoleillement (paramètre 10) - Agrégation par jour
    sun_files = [f for f in os.listdir('./habo_plast/smhi_data_2022-today/parameter_10') if f.endswith('.csv')]
    sun_dfs = []
    
    for file in sun_files:
        try:
            df = pd.read_csv(f'./habo_plast/smhi_data_2022-today/parameter_10/{file}')
            df['station'] = file.split('-')[0].replace('station_', '')
            sun_dfs.append(df)
        except:
            print(f"Problème avec le fichier ensoleillement: {file}")
    
    sun_df = pd.concat(sun_dfs, ignore_index=True)
    sun_df['Date'] = pd.to_datetime(sun_df['Datum'])
    
    # Convertir en heures si en minutes et agréger par jour
    sun_df['Solskenstid'] = sun_df['Solskenstid'] / 60  # Convertir en heures si en minutes
    sun_daily = sun_df.groupby(['Date', 'station'])['Solskenstid'].sum().reset_index()
    sun_daily = sun_daily.groupby('Date')['Solskenstid'].mean().reset_index()
    sun_daily.rename(columns={'Solskenstid': 'SunshineHours'}, inplace=True)
    
    # Fusionner toutes les données météo
    weather_df = temp_daily.merge(precip_daily, on='Date', how='outer')
    weather_df = weather_df.merge(snow_daily, on='Date', how='outer')
    weather_df = weather_df.merge(sun_daily, on='Date', how='outer')
    
    # Convertir la date au même format que les données d'électricité
    weather_df['Day'] = weather_df['Date'].dt.date
    weather_df = weather_df.drop('Date', axis=1)
    
    return weather_df

def prepare_data():
    # Fusionner les données d'électricité et météo
    print("Chargement des données d'électricité...")
    electricity_data = load_electricity_data()
    print("Chargement des données météorologiques...")
    weather_data = load_weather_data()

    print("Fusion des données...")
    merged_data = electricity_data.merge(weather_data, on='Day', how='inner')

    # Gérer les valeurs manquantes
    print("Traitement des valeurs manquantes...")
    merged_data = merged_data.fillna(method='ffill')  # Forward fill pour les données manquantes
    merged_data = merged_data.dropna()  # Supprimer les lignes restantes avec des valeurs manquantes

    # Convertir 'Day' en datetime et extraire les caractéristiques de date
    print("Création de caractéristiques supplémentaires...")
    merged_data['Day'] = pd.to_datetime(merged_data['Day'])
    merged_data['Month'] = merged_data['Day'].dt.month
    merged_data['DayOfWeek'] = merged_data['Day'].dt.dayofweek
    merged_data['Season'] = merged_data['Day'].dt.month % 12 // 3 + 1  # 1: Hiver, 2: Printemps, 3: Été, 4: Automne

    # Variables retardées (prix des jours précédents)
    for i in range(1, 8):  # Ajouter les prix des 7 jours précédents
        merged_data[f'Price_lag_{i}'] = merged_data['Price'].shift(i)

    # Variables moyennes mobiles
    merged_data['Price_MA7'] = merged_data['Price'].rolling(window=7).mean()
    merged_data['Temp_MA7'] = merged_data['Temperature'].rolling(window=7).mean()

    # Supprimer les premières lignes qui ont des NaN en raison des variables retardées
    merged_data = merged_data.dropna()

    # Explorer les corrélations - EXCLURE les colonnes non numériques
    print("Analyse des corrélations...")
    # Sélectionner uniquement les colonnes numériques pour la corrélation
    numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
    correlation_matrix = merged_data[numeric_columns].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Corrélation entre les variables')
    plt.savefig('./results/correlation_matrix.png')
    plt.close()

    # Préparation des données pour les modèles
    # Exclure 'Day' qui est maintenant un datetime
    X = merged_data.drop(['Day', 'Price'], axis=1)
    # S'assurer que toutes les colonnes sont numériques
    X = X.select_dtypes(include=[np.number])
    y = merged_data['Price']

    # Normalisation des données
    print("Normalisation des données...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Diviser les données en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Créer le répertoire results s'il n'existe pas
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    # Créer le répertoire models s'il n'existe pas
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Sauvegarde pour utilisation ultérieure
    print("Sauvegarde des données préparées...")
    # Créer le répertoire data s'il n'existe pas
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    np.save('./data/X_train.npy', X_train)
    np.save('./data/X_test.npy', X_test)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_test.npy', y_test)
    
    # Sauvegarde du scaler pour une utilisation future
    joblib.dump(scaler, './models/scaler.pkl')

    # Sauvegarde des noms de colonnes pour référence
    with open('./data/feature_names.txt', 'w') as f:
        for column in X.columns:
            f.write(f"{column}\n")
    
    # Sauvegarde des données fusionnées pour référence
    # Convertir Day en string pour éviter les problèmes de sérialisation
    merged_data_to_save = merged_data.copy()
    merged_data_to_save['Day'] = merged_data_to_save['Day'].dt.strftime('%Y-%m-%d')
    merged_data_to_save.to_csv('./data/merged_data.csv', index=False)
    
    print("Préparation des données terminée.")
    return X_train, X_test, y_train, y_test, X.columns

if __name__ == "__main__":
    # Créer le répertoire data s'il n'existe pas
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    prepare_data()