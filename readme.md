# Electricity Price Prediction for Habo Plast

This project develops three machine learning models to predict daily electricity prices in the SE3 zone of Sweden, using weather and historical data.

## Project Structure
habo_plast_electricity_prediction/
│
├── data/                           # Folder for processed data
│
├── models/                         # Folder for trained models
│
├── results/                        # Folder for graphs and results
│
├── data_preparation.py             # Data preparation script
├── model1_linear_regression.py     # Linear regression model script
├── model2_ensemble_methods.py      # Ensemble methods script
├── model3_neural_networks.py       # Neural networks script
├── model_comparison.py             # Model comparison script
├── prediction_function.py          # Prediction function
├── main.py                         # Main script
└── journals/                       # Individual project journals

## Installation and Requirements

To run this project, you will need the following Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
```

## Required Data
The project uses the following data which must be placed in the appropriate directory:

Electricity price data: ENTSO-E price CSV files for SE3 (2016-2024) in ./habo_plast/electricity/
SMHI weather data: CSV files from stations in ./habo_plast/smhi_data_2022-today/

## Running the Project
To run the complete project:

```bash
python main.py
```

This will run in sequence:

- Data preparation
- Training of the three models
- Model comparison
- Generation of visualizations and results

## Prediction with Trained Models
To use the trained models for a prediction:

```bash
from prediction_function import predict_electricity_price
from datetime import datetime

# Usage example
prediction = predict_electricity_price(
    date=datetime.now(),
    temperature=15.0,          # °C
    precipitation=2.5,         # mm
    snow_depth=0.0,            # cm
    sunshine_hours=8.0,        # hours
    historical_prices=[45.2, 47.8, 50.1, 49.3, 48.7, 51.2, 52.5]  # last 7 days
)

print(prediction)
```

## Results
The models developed in this project achieved the following performance on the test set:

Model|R²|RMSE (EUR/MWh)|MAE (EUR/MWh)
-|-|-|-
Ridge|0.9987|1.7242|0.9641
Random Forest|0.8152|20.9195|10.5179
MLP|0.9755|7.6231|4.8921
Ensemble (average)|0.9633|9.3286|5.0203

## Contributors

Rémi Maigrot
Mathieu Rio
Arthur Bourdin
Deepa Krishnan
Loshma Latha Babu

## License
This project is developed in the academic framework of Jönköping University. The data used is the property of their respective providers (ENTSO-E and SMHI).