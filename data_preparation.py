import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load electricity price data
def load_electricity_data():
    # Choose files from the SE3 zone (since weather data concerns SE3)
    electricity_files = [f for f in os.listdir('./habo_plast/electricity') if f.startswith('SE3_') and f.endswith('entsoe.csv')]
    
    print(f"Files found: {electricity_files}")
    
    dfs = []
    for file in electricity_files:
        try:
            df = pd.read_csv(f'./habo_plast/electricity/{file}')
            # Examine the file structure
            if dfs == []:  # only for the first file
                print(f"First lines of {file}:")
                print(df.head())
                print(f"Data types: {df.dtypes}")
            
            # Make sure the price column is numeric
            if 'Day-ahead Price [EUR/MWh]' in df.columns:
                # Check if the column contains non-numeric values
                try:
                    df['Day-ahead Price [EUR/MWh]'] = pd.to_numeric(df['Day-ahead Price [EUR/MWh]'], errors='coerce')
                    print(f"Successful numeric conversion for {file}")
                except Exception as e:
                    print(f"Error during numeric conversion: {e}")
                    # Display some problematic values
                    print("Examples of values in the price column:")
                    print(df['Day-ahead Price [EUR/MWh]'].iloc[:5].tolist())
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    # Concatenate all files
    electricity_df = pd.concat(dfs, ignore_index=True)
    
    # Convert dates
    try:
        # If the format is "DD.MM.YYYY HH:MM - DD.MM.YYYY HH:MM"
        electricity_df['Date'] = electricity_df['MTU (CET/CEST)'].apply(
            lambda x: pd.to_datetime(x.split(' - ')[0], format='%d.%m.%Y %H:%M')
        )
    except Exception as e:
        print(f"Error converting dates: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    
    # Aggregate by day to match daily weather data
    electricity_df['Day'] = electricity_df['Date'].dt.date
    
    # Make sure the price column is numeric before aggregation
    print("Final verification of data types before aggregation:")
    print(electricity_df.dtypes)
    
    # Use a more robust method for aggregation
    try:
        # Filter non-numeric values
        electricity_df = electricity_df[pd.to_numeric(electricity_df['Day-ahead Price [EUR/MWh]'], errors='coerce').notna()]
        electricity_df['Day-ahead Price [EUR/MWh]'] = pd.to_numeric(electricity_df['Day-ahead Price [EUR/MWh]'])
        
        # Group and calculate the average
        daily_electricity = electricity_df.groupby('Day')['Day-ahead Price [EUR/MWh]'].mean().reset_index()
        daily_electricity.rename(columns={'Day-ahead Price [EUR/MWh]': 'Price'}, inplace=True)
    except Exception as e:
        print(f"Error during aggregation: {e}")
        
        # Alternative: manually process each group
        unique_days = electricity_df['Day'].unique()
        prices = []
        
        for day in unique_days:
            day_data = electricity_df[electricity_df['Day'] == day]
            try:
                # Convert to numeric and ignore non-convertible values
                numeric_prices = pd.to_numeric(day_data['Day-ahead Price [EUR/MWh]'], errors='coerce')
                avg_price = numeric_prices.mean()
                prices.append({'Day': day, 'Price': avg_price})
            except Exception as e_inner:
                print(f"Error for day {day}: {e_inner}")
        
        daily_electricity = pd.DataFrame(prices)
    
    print("Daily price data processed successfully.")
    return daily_electricity

# Load weather data
def load_weather_data():
    # Temperature (parameter 2)
    temp_files = [f for f in os.listdir('./habo_plast/smhi_data_2022-today/parameter_2') if f.endswith('.csv')]
    temp_dfs = []
    
    for file in temp_files:
        try:
            df = pd.read_csv(f'./habo_plast/smhi_data_2022-today/parameter_2/{file}')
            df['station'] = file.split('-')[0].replace('station_', '')
            temp_dfs.append(df)
        except:
            print(f"Problem with temperature file: {file}")
    
    temp_df = pd.concat(temp_dfs, ignore_index=True)
    temp_df['Date'] = pd.to_datetime(temp_df['Datum'])
    
    # Calculate average daily temperature across all stations
    temp_daily = temp_df.groupby('Date')['Lufttemperatur'].mean().reset_index()
    temp_daily.rename(columns={'Lufttemperatur': 'Temperature'}, inplace=True)
    
    # Precipitation (parameter 5)
    precip_files = [f for f in os.listdir('./habo_plast/smhi_data_2022-today/parameter_5') if f.endswith('.csv')]
    precip_dfs = []
    
    for file in precip_files:
        try:
            df = pd.read_csv(f'./habo_plast/smhi_data_2022-today/parameter_5/{file}')
            df['station'] = file.split('-')[0].replace('station_', '')
            precip_dfs.append(df)
        except:
            print(f"Problem with precipitation file: {file}")
    
    precip_df = pd.concat(precip_dfs, ignore_index=True)
    precip_df['Date'] = pd.to_datetime(precip_df['Datum'])
    
    # Calculate average daily precipitation across all stations
    precip_daily = precip_df.groupby('Date')['Nederbördsmängd'].mean().reset_index()
    precip_daily.rename(columns={'Nederbördsmängd': 'Precipitation'}, inplace=True)
    
    # Snow depth (parameter 8)
    snow_files = [f for f in os.listdir('./habo_plast/smhi_data_2022-today/parameter_8') if f.endswith('.csv')]
    snow_dfs = []
    
    for file in snow_files:
        try:
            df = pd.read_csv(f'./habo_plast/smhi_data_2022-today/parameter_8/{file}')
            df['station'] = file.split('-')[0].replace('station_', '')
            snow_dfs.append(df)
        except:
            print(f"Problem with snow file: {file}")
    
    snow_df = pd.concat(snow_dfs, ignore_index=True)
    snow_df['Date'] = pd.to_datetime(snow_df['Datum'])
    
    # Calculate average daily snow depth across all stations
    snow_daily = snow_df.groupby('Date')['Snödjup'].mean().reset_index()
    snow_daily.rename(columns={'Snödjup': 'SnowDepth'}, inplace=True)
    
    # Sunshine duration (parameter 10) - Aggregation by day
    sun_files = [f for f in os.listdir('./habo_plast/smhi_data_2022-today/parameter_10') if f.endswith('.csv')]
    sun_dfs = []
    
    for file in sun_files:
        try:
            df = pd.read_csv(f'./habo_plast/smhi_data_2022-today/parameter_10/{file}')
            df['station'] = file.split('-')[0].replace('station_', '')
            sun_dfs.append(df)
        except:
            print(f"Problem with sunshine file: {file}")
    
    sun_df = pd.concat(sun_dfs, ignore_index=True)
    sun_df['Date'] = pd.to_datetime(sun_df['Datum'])
    
    # Convert to hours if in minutes and aggregate by day
    sun_df['Solskenstid'] = sun_df['Solskenstid'] / 60  # Convert to hours if in minutes
    sun_daily = sun_df.groupby(['Date', 'station'])['Solskenstid'].sum().reset_index()
    sun_daily = sun_daily.groupby('Date')['Solskenstid'].mean().reset_index()
    sun_daily.rename(columns={'Solskenstid': 'SunshineHours'}, inplace=True)
    
    # Merge all weather data
    weather_df = temp_daily.merge(precip_daily, on='Date', how='outer')
    weather_df = weather_df.merge(snow_daily, on='Date', how='outer')
    weather_df = weather_df.merge(sun_daily, on='Date', how='outer')
    
    # Convert date to the same format as electricity data
    weather_df['Day'] = weather_df['Date'].dt.date
    weather_df = weather_df.drop('Date', axis=1)
    
    return weather_df

def prepare_data():
    # Merge electricity and weather data
    print("Loading electricity data...")
    electricity_data = load_electricity_data()
    print("Loading weather data...")
    weather_data = load_weather_data()

    print("Merging data...")
    merged_data = electricity_data.merge(weather_data, on='Day', how='inner')

    # Handle missing values
    print("Processing missing values...")
    merged_data = merged_data.fillna(method='ffill')  # Forward fill for missing data
    merged_data = merged_data.dropna()  # Remove remaining rows with missing values

    # Convert 'Day' to datetime and extract date features
    print("Creating additional features...")
    merged_data['Day'] = pd.to_datetime(merged_data['Day'])
    merged_data['Month'] = merged_data['Day'].dt.month
    merged_data['DayOfWeek'] = merged_data['Day'].dt.dayofweek
    merged_data['Season'] = merged_data['Day'].dt.month % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall

    # Lagged variables (prices from previous days)
    for i in range(1, 8):  # Add prices from the previous 7 days
        merged_data[f'Price_lag_{i}'] = merged_data['Price'].shift(i)

    # Moving average variables
    merged_data['Price_MA7'] = merged_data['Price'].rolling(window=7).mean()
    merged_data['Temp_MA7'] = merged_data['Temperature'].rolling(window=7).mean()

    # Remove first rows that have NaN due to lagged variables
    merged_data = merged_data.dropna()

    # Explore correlations - EXCLUDE non-numeric columns
    print("Analyzing correlations...")
    # Select only numeric columns for correlation
    numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
    correlation_matrix = merged_data[numeric_columns].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation between variables')
    plt.savefig('./results/correlation_matrix.png')
    plt.close()

    # Data preparation for models
    # Exclude 'Day' which is now a datetime
    X = merged_data.drop(['Day', 'Price'], axis=1)
    # Make sure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    y = merged_data['Price']

    # Data normalization
    print("Normalizing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create the results directory if it doesn't exist
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    # Create the models directory if it doesn't exist
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # Save for later use
    print("Saving prepared data...")
    # Create the data directory if it doesn't exist
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    np.save('./data/X_train.npy', X_train)
    np.save('./data/X_test.npy', X_test)
    np.save('./data/y_train.npy', y_train)
    np.save('./data/y_test.npy', y_test)
    
    # Save the scaler for future use
    joblib.dump(scaler, './models/scaler.pkl')

    # Save column names for reference
    with open('./data/feature_names.txt', 'w') as f:
        for column in X.columns:
            f.write(f"{column}\n")
    
    # Save the merged data for reference
    # Convert Day to string to avoid serialization issues
    merged_data_to_save = merged_data.copy()
    merged_data_to_save['Day'] = merged_data_to_save['Day'].dt.strftime('%Y-%m-%d')
    merged_data_to_save.to_csv('./data/merged_data.csv', index=False)
    
    print("Data preparation completed.")
    return X_train, X_test, y_train, y_test, X.columns

if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    prepare_data()
