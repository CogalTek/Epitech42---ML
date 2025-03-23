# User Manual - Electricity Price Prediction System for Habo Plast

This manual explains how to use the electricity price prediction system developed for Habo Plast. The system allows you to predict daily electricity prices in the SE3 zone of Sweden using weather and historical data.

## Installation

1. Make sure Python 3.8 or higher is installed on your system.
2. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
3. Clone or download the repository containing the source code.

## File Structure
The system is made up of the following files:

- data_preparation.py: Data preparation module
- model1_linear_regression.py: Implementation of the Ridge regression model
- model2_ensemble_methods.py: Implementation of ensemble methods (Random Forest)
- model3_neural_networks.py: Implementation of neural networks
- model_comparison.py: Model comparison scripts
- prediction_function.py: Unified function for making predictions
- main.py: Main script to run the entire pipeline

## Running the Complete Pipeline
To run the entire pipeline (data preparation, model training, evaluation and comparison):

```bash
python main.py
```

This command:

1) Will prepare the data from raw sources
2) Will train the three prediction models
3) Will evaluate and compare their performance
4) Will generate visualizations and reports in the results/ folder

The complete process usually takes about 10-15 minutes depending on your computer's performance.

## Making New Predictions
To use the trained models to predict electricity prices for a new day, use the predict_electricity_price function from the prediction_function.py module:

```bash
from prediction_function import predict_electricity_price
from datetime import datetime

# Example prediction for a specific day
prediction = predict_electricity_price(
    date=datetime(2025, 3, 15),             # Prediction date
    temperature=10.5,                       # Average temperature (°C)
    precipitation=1.2,                      # Precipitation (mm)
    snow_depth=0.0,                         # Snow depth (cm)
    sunshine_hours=6.5,                     # Sunshine duration (hours)
    historical_prices=[45.2, 47.8, 50.1, 49.3, 48.7, 51.2, 52.5]  # Prices for the last 7 days
)

print(prediction)
```

This function returns a dictionary containing the predictions of each model as well as the ensemble prediction:

```bash
    'Linear Regression': 53.24,         # Ridge model prediction
    'Ensemble Method': 55.12,           # Random Forest prediction
    'Neural Network': 54.18,            # MLP prediction
    'Ensemble (Average)': 54.18         # Average of the three models
```

## Understanding the Results
To understand the prediction results:

1) The Linear Regression (Ridge) model showed the best overall performance (R² = 0.9987) and should be considered the most reliable in most situations.
2) The Neural Network (MLP) model is also very accurate (R² = 0.9755) and can sometimes better capture sudden price changes.
3) The Ensemble (Average) prediction combines the three models and can offer extra protection against large errors from any single model.
4) Use the MAE (Mean Absolute Error) value to estimate the typical error of the predictions. For the Ridge model, the average error is about 0.96 EUR/MWh.

## Model Updates
To keep prediction accuracy high, it is recommended to retrain the models regularly (ideally every month) with new price and weather data. To do this:

1) Update the CSV files in the ./habo_plast/electricity/ and ./habo_plast/smhi_data_2022-today/ folders
2) Run the main script again:

```bash
python main.py
```

## Troubleshooting
If you encounter problems:

1) Error when loading data: Check that the CSV files are in the expected paths and have the correct format.
2) Error when making predictions: Make sure the models have been properly trained and that the model files are present in the ./models/ folder.
3) Abnormal predictions: If the predictions seem unrealistic (too high or too low), check that the input values (temperature, precipitation, etc.) are in reasonable ranges and that the historical prices are correctly ordered (from oldest to newest).

For any additional help, contact the development team.