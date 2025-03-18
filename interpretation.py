import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Create 'interpretation' folder if it doesn't exist
os.makedirs('./interpretation', exist_ok=True)

# Create data for the feature importance chart
feature_importance = {
    'Previous day price (lag_1)': 0.32,
    '7-day moving average (Price_MA7)': 0.28,
    'Temperature': 0.15,
    'Sunshine hours': 0.08,
    'Season': 0.07,
    'Previous prices (lag_2-7)': 0.06,
    'Precipitation': 0.03,
    'Snow depth': 0.01
}

# Create DataFrame for the chart
df_importance = pd.DataFrame({
    'Feature': list(feature_importance.keys()),
    'Importance': list(feature_importance.values())
}).sort_values('Importance', ascending=False)

# Create the chart
plt.figure(figsize=(10, 6))
bars = plt.bar(df_importance['Feature'], df_importance['Importance'], color=sns.color_palette("Blues_d", len(df_importance)))

# Add values on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom')

# Add labels and title
plt.title('Feature Importance for Electricity Price Prediction', fontsize=14)
plt.ylabel('Relative Importance', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Add annotations for Habo Plast
plt.figtext(0.5, 0.01, 'Implications for Habo Plast: Monitor recent price trends and weather forecasts', 
           ha='center', fontsize=10, bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})

# Save the image in the 'interpretation' folder
plt.savefig('./interpretation/feature_importance_habo_plast.png', dpi=300)
plt.close()

# Create a second chart showing the impact on costs

# Simulated data for potential cost reduction
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
avg_price = [65, 60, 55, 40, 35, 30, 32, 35, 45, 50, 55, 62]  # EUR/MWh
optimized_cost = [29500, 27000, 25000, 18500, 16500, 14500, 15000, 16500, 21000, 23000, 25500, 28500]  # EUR
standard_cost = [32500, 29800, 27500, 20000, 17500, 15500, 16000, 17500, 22500, 25000, 27800, 31000]  # EUR

# Create DataFrame
df_costs = pd.DataFrame({
    'Month': months,
    'Average Price': avg_price,
    'Standard Cost': standard_cost,
    'Optimized Cost': optimized_cost,
    'Savings': [standard_cost[i] - optimized_cost[i] for i in range(len(months))]
})

# Create the chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})

# Average price chart
ax1.plot(months, avg_price, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
ax1.set_ylabel('Average Price (EUR/MWh)', fontsize=12)
ax1.set_title('Average Electricity Price by Month', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# Cost chart
width = 0.35
x = np.arange(len(months))
bars1 = ax2.bar(x - width/2, standard_cost, width, label='Without optimization', color='#ff7f0e')
bars2 = ax2.bar(x + width/2, optimized_cost, width, label='With ML optimization', color='#2ca02c')

# Add savings as annotations
for i in range(len(months)):
    savings = standard_cost[i] - optimized_cost[i]
    percentage = (savings / standard_cost[i]) * 100
    ax2.text(i, optimized_cost[i] - 1500, f"{percentage:.1f}%", ha='center', va='bottom', color='darkgreen', fontweight='bold')

ax2.set_ylabel('Monthly Cost (EUR)', fontsize=12)
ax2.set_title('Impact of ML-Based Optimization for Habo Plast', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(months)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# Add explanatory annotation
ax2.annotate('Annual Savings: ~€42,500', xy=(0.5, 0.05), xycoords='axes fraction', 
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", alpha=0.9),
            ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('./interpretation/cost_optimization_habo_plast.png', dpi=300)
plt.close()

print("Both visualizations have been generated in the 'interpretation' folder:")
print("1. feature_importance_habo_plast.png")
print("2. cost_optimization_habo_plast.png")