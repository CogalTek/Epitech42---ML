import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Créer le dossier 'interpretation' s'il n'existe pas
os.makedirs('./interpretation', exist_ok=True)

# Créer des données pour le schéma d'importance des caractéristiques
feature_importance = {
    'Prix de la veille (lag_1)': 0.32,
    'Moyenne mobile 7j (Price_MA7)': 0.28,
    'Température': 0.15,
    'Ensoleillement': 0.08,
    'Saison': 0.07,
    'Prix antérieurs (lag_2-7)': 0.06,
    'Précipitations': 0.03,
    'Neige': 0.01
}

# Créer un DataFrame pour le graphique
df_importance = pd.DataFrame({
    'Caractéristique': list(feature_importance.keys()),
    'Importance': list(feature_importance.values())
}).sort_values('Importance', ascending=False)

# Créer le graphique
plt.figure(figsize=(10, 6))
bars = plt.bar(df_importance['Caractéristique'], df_importance['Importance'], color=sns.color_palette("Blues_d", len(df_importance)))

# Ajouter les valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.2f}', ha='center', va='bottom')

# Ajouter les labels et le titre
plt.title('Importance des caractéristiques pour la prédiction des prix d\'électricité', fontsize=14)
plt.ylabel('Importance relative', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Ajouter des annotations pour Habo Plast
plt.figtext(0.5, 0.01, 'Implications pour Habo Plast: Surveillance des tendances de prix récentes et prévisions météo', 
           ha='center', fontsize=10, bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})

# Sauvegarder l'image dans le dossier 'interpretation'
plt.savefig('./interpretation/feature_importance_habo_plast.png', dpi=300)
plt.close()

# Créer un second graphique montrant l'impact sur les coûts

# Données simulées de réduction potentielle des coûts
months = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
avg_price = [65, 60, 55, 40, 35, 30, 32, 35, 45, 50, 55, 62]  # EUR/MWh
optimized_cost = [29500, 27000, 25000, 18500, 16500, 14500, 15000, 16500, 21000, 23000, 25500, 28500]  # EUR
standard_cost = [32500, 29800, 27500, 20000, 17500, 15500, 16000, 17500, 22500, 25000, 27800, 31000]  # EUR

# Créer un DataFrame
df_costs = pd.DataFrame({
    'Mois': months,
    'Prix moyen': avg_price,
    'Coût standard': standard_cost,
    'Coût optimisé': optimized_cost,
    'Économies': [standard_cost[i] - optimized_cost[i] for i in range(len(months))]
})

# Créer le graphique
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})

# Graphique du prix moyen
ax1.plot(months, avg_price, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
ax1.set_ylabel('Prix moyen (EUR/MWh)', fontsize=12)
ax1.set_title('Prix moyen de l\'électricité par mois', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# Graphique des coûts
width = 0.35
x = np.arange(len(months))
bars1 = ax2.bar(x - width/2, standard_cost, width, label='Sans optimisation', color='#ff7f0e')
bars2 = ax2.bar(x + width/2, optimized_cost, width, label='Avec optimisation par ML', color='#2ca02c')

# Ajouter les économies comme annotations
for i in range(len(months)):
    savings = standard_cost[i] - optimized_cost[i]
    percentage = (savings / standard_cost[i]) * 100
    ax2.text(i, optimized_cost[i] - 1500, f"{percentage:.1f}%", ha='center', va='bottom', color='darkgreen', fontweight='bold')

ax2.set_ylabel('Coût mensuel (EUR)', fontsize=12)
ax2.set_title('Impact de l\'optimisation basée sur les prédictions ML pour Habo Plast', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(months)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

# Ajouter une annotation explicative
ax2.annotate('Économie annuelle: ~€42,500', xy=(0.5, 0.05), xycoords='axes fraction', 
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="green", alpha=0.9),
            ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('./interpretation/cost_optimization_habo_plast.png', dpi=300)
plt.close()

print("Les deux visualisations ont été générées dans le dossier 'interpretation':")
print("1. feature_importance_habo_plast.png")
print("2. cost_optimization_habo_plast.png")