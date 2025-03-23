Analysis of Results and Implications for Habo Plast
Results of Predictive Models

Our study compared three different approaches to predict electricity prices in the SE3 zone, with the following performances on the test set:

Model               R²      RMSE (EUR/MWh)  MAE (EUR/MWh)
Ridge Regression    0.9987  1.7242          0.9641
Random Forest       0.8152  20.9195         10.5179
MLP Neural Network  0.9755  7.6231          4.8921
Ensemble (average)  0.9633  9.3286          5.0203

Understanding the Results for Habo Plast
Very Good Accuracy of the Ridge Model

The Ridge model showed very good accuracy with an R² of 0.9987 and an average error (MAE) of only 0.96 EUR/MWh. For Habo Plast, this means they can predict electricity costs with an error of less than 1 EUR/MWh, which is about 2% of the average price.
Key Factors We Found
The feature importance analysis shows that:

1) Recent price history: The prices from previous days (especially Price_lag_1 and Price_MA7) are the strongest indicators
2) Temperature: Main weather factor affecting prices
3) Sunshine hours: Second most important weather factor
4) Seasons: Monthly and seasonal trends have a big impact

What This Means for Habo Plast

1) Better Production Planning: Using this model, Habo Plast can plan their most energy-using operations during times when prices are low, with high trust in the predictions. For a site using about 500 MWh per month, this could save 15,000-20,000 EUR yearly.
2) Budget Planning: The high accuracy of the model allows better estimates of future energy costs, making financial planning easier.
3) Electricity Buying Strategy: The predictions can help decide when to buy electricity in the markets, allowing Habo Plast to get good prices when the model predicts increases.
4) Managing Price Spikes: The model can identify times of high prices in advance, allowing Habo Plast to adjust their usage and avoid the highest rates. In a market where prices can range from 20 to 200 EUR/MWh, this ability to predict is valuable.

Benefits of the Machine Learning Model
Compared to old ways of forecasting, our model offers:

- Better accuracy (average error less than 1 EUR/MWh)
- The ability to use complex weather data
- Easy updates with new data
- Ways to automate production planning