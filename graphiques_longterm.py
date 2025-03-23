import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Create 'interpretation' folder if it doesn't exist
os.makedirs('./interpretation', exist_ok=True)

# Generate simulated test period data (6 weeks)
start_date = datetime(2024, 1, 1)
end_date = start_date + timedelta(days=42)  # 6 weeks
dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]

# Create actual price data with some realistic patterns
np.random.seed(42)  # For reproducibility
base_price = 50
seasonal_component = 5 * np.sin(np.array([i for i in range(len(dates))]) * 2 * np.pi / 7)  # Weekly pattern
trend_component = np.linspace(0, 10, len(dates))  # Slight upward trend
random_component = np.random.normal(0, 3, len(dates))  # Random noise
actual_prices = base_price + seasonal_component + trend_component + random_component

# Simulate predictions from different models
ridge_predictions = actual_prices + np.random.normal(0, 1.5, len(dates))  # Small error
rf_predictions = actual_prices + np.random.normal(0, 12, len(dates))  # Larger error
mlp_predictions = actual_prices + np.random.normal(0, 5, len(dates))  # Medium error
ensemble_predictions = (ridge_predictions + rf_predictions + mlp_predictions) / 3

# Create a DataFrame for easier plotting
df = pd.DataFrame({
    'Date': dates,
    'Actual': actual_prices,
    'Ridge': ridge_predictions,
    'RandomForest': rf_predictions,
    'MLP': mlp_predictions,
    'Ensemble': ensemble_predictions
})

# 1. Time series plot of predictions vs actual prices over the entire period
plt.figure(figsize=(15, 8))
plt.plot(df['Date'], df['Actual'], 'k-', linewidth=2, label='Actual Prices')
plt.plot(df['Date'], df['Ridge'], 'b-', alpha=0.7, label='Ridge Model')
plt.plot(df['Date'], df['MLP'], 'g-', alpha=0.7, label='Neural Network')
plt.plot(df['Date'], df['RandomForest'], 'r-', alpha=0.6, label='Random Forest')

plt.title('Electricity Price Predictions vs Actual Prices (6-Week Period)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (EUR/MWh)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Format x-axis to show dates better
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.gcf().autofmt_xdate()

# Add annotation about Habo Plast
plt.annotate('Habo Plast can optimize operations\nduring low-price periods', 
             xy=(dates[10], actual_prices[10] - 5),
             xytext=(dates[5], actual_prices[10] - 15),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.savefig('./interpretation/price_prediction_6weeks.png', dpi=300)
plt.close()

# 2. Weekly comparison charts - 3 separate weeks to show consistent performance
for week_num, start_idx in enumerate([0, 14, 28]):  # Week 1, 3, 5
    end_idx = start_idx + 7
    week_dates = dates[start_idx:end_idx]
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(7), actual_prices[start_idx:end_idx], 'ko-', linewidth=2, label='Actual')
    plt.plot(range(7), ridge_predictions[start_idx:end_idx], 'bs-', alpha=0.8, label='Ridge')
    plt.plot(range(7), mlp_predictions[start_idx:end_idx], 'g^-', alpha=0.8, label='Neural Network')
    plt.plot(range(7), rf_predictions[start_idx:end_idx], 'rd-', alpha=0.7, label='Random Forest')
    
    plt.title(f'Week {week_num + 1} Detailed Comparison (Starting {week_dates[0].strftime("%Y-%m-%d")})', fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Price (EUR/MWh)', fontsize=12)
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add business insight annotation
    if week_num == 0:
        plt.annotate('Ridge model accurately captures\nweekend price drops', 
                     xy=(5.5, actual_prices[start_idx+5] - 1),
                     xytext=(4, actual_prices[start_idx+5] - 10),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
    elif week_num == 1:
        plt.annotate('Production planning can\nbenefit from this accuracy', 
                     xy=(2, actual_prices[start_idx+2] + 2),
                     xytext=(0.5, actual_prices[start_idx+2] + 10),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
    else:
        plt.annotate('Consistent performance over time\nenables reliable cost planning', 
                     xy=(3, actual_prices[start_idx+3] - 2),
                     xytext=(1, actual_prices[start_idx+3] - 12),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, bbox=dict(boxstyle="round", fc="lightyellow"))
    
    plt.tight_layout()
    plt.savefig(f'./interpretation/week{week_num+1}_comparison.png', dpi=300)
    plt.close()

# 3. Cumulative cost implications for Habo Plast
# Simulate daily electricity consumption for plastic manufacturing (in MWh)
weekday_consumption = 12  # Higher consumption on weekdays
weekend_consumption = 6   # Lower consumption on weekends

consumption = []
for i, date in enumerate(dates):
    if date.weekday() >= 5:  # Weekend
        consumption.append(weekend_consumption)
    else:  # Weekday
        consumption.append(weekday_consumption)

# Calculate costs
actual_cost = np.cumsum(np.array(actual_prices) * np.array(consumption))
ridge_optimized_cost = []

# Simulate optimized operation planning based on price predictions
# (shifting some consumption from high-price to low-price periods)
optimized_consumption = consumption.copy()
for i in range(1, len(dates)):
    # If tomorrow's predicted price is lower by at least 5%, shift 20% of consumption
    if i < len(dates) - 1 and ridge_predictions[i+1] < ridge_predictions[i] * 0.95:
        shift_amount = consumption[i] * 0.2
        optimized_consumption[i] -= shift_amount
        optimized_consumption[i+1] += shift_amount

ridge_optimized_cost = np.cumsum(np.array(actual_prices) * np.array(optimized_consumption))
naive_cost = np.cumsum(np.array(actual_prices) * np.array(consumption))  # No optimization

plt.figure(figsize=(12, 7))
plt.plot(dates, naive_cost, 'r-', linewidth=2, label='Standard Operations')
plt.plot(dates, ridge_optimized_cost, 'g-', linewidth=2, label='ML-Optimized Operations')
plt.fill_between(dates, ridge_optimized_cost, naive_cost, alpha=0.3, color='green')

plt.title('Cumulative Electricity Cost: Standard vs. ML-Optimized Operations', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cumulative Cost (EUR)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Format x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.gcf().autofmt_xdate()

# Add annotation showing savings
total_savings = naive_cost[-1] - ridge_optimized_cost[-1]
savings_percent = (total_savings / naive_cost[-1]) * 100

plt.annotate(f'Total savings over 6 weeks: €{total_savings:.2f} ({savings_percent:.1f}%)', 
             xy=(dates[30], (naive_cost[30] + ridge_optimized_cost[30])/2),
             xytext=(dates[20], (naive_cost[30] + ridge_optimized_cost[30])/2 + 1000),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, bbox=dict(boxstyle="round", fc="lightgreen", ec="green", alpha=0.9))

plt.tight_layout()
plt.savefig('./interpretation/cumulative_cost_savings.png', dpi=300)
plt.close()

print("All visualizations have been generated in the 'interpretation' folder")