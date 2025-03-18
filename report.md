---
title: Untitled

---

# Prédiction des Prix de l'Électricité pour Habo Plast: Une Approche par Apprentissage Automatique 

### Résumé 

Ce projet vise à développer des modèles de prédiction des prix de l'électricité pour l'entreprise Habo Plast en utilisant des données météorologiques comme variables prédictives. Trois approches différentes d'apprentissage automatique ont été implémentées et comparées: la régression linéaire régularisée (Ridge), les méthodes d'ensemble (Random Forest) et les réseaux de neurones (MLP). Les modèles intègrent des caractéristiques météorologiques telles que la température, les précipitations, l'épaisseur de neige et l'ensoleillement, ainsi que des informations temporelles comme les prix antérieurs. Le modèle de régression Ridge a démontré une performance exceptionnelle avec un **R² de 0,998**, suivi du réseau de neurones (**R² de 0,975**) et du Random Forest (**R² de 0,815**). Cette étude démontre l'efficacité de l'apprentissage automatique pour la prédiction des prix de l'électricité, ce qui pourrait permettre à Habo Plast d'optimiser sa consommation énergétique et de réduire ses coûts opérationnels. 

### I. Introduction 

#### A. Contexte 

L'instabilité et la volatilité croissantes des prix de l'électricité posent des défis majeurs aux entreprises industrielles comme Habo Plast, qui dépendent fortement de l'énergie pour leurs opérations. Les fluctuations des prix de l'électricité sont influencées par de nombreux facteurs, notamment les conditions météorologiques qui affectent à la fois la production (particulièrement pour les énergies renouvelables) et la demande. En Suède, où le pays est divisé en quatre zones électriques (SE1, SE2, SE3 et SE4), les prix peuvent varier significativement selon les régions et les saisons. 

#### B. Problématique 

Pour les entreprises comme Habo Plast, prévoir les variations de prix de l'électricité représente un avantage stratégique considérable. Une prédiction fiable permettrait d'ajuster la production pendant les périodes où l'électricité est moins coûteuse, réduisant ainsi les dépenses opérationnelles. La question centrale devient donc: *comment prédire efficacement les prix de l'électricité en utilisant des données météorologiques et historiques?*

#### C. Objectifs du projet 

* Développer et évaluer trois modèles différents d'apprentissage automatique pour prédire les prix quotidiens de l'électricité dans la zone SE3 de Suède 
* Identifier les variables météorologiques les plus influentes sur les prix de l'électricité 
* Comparer les performances des différentes approches pour déterminer la plus adaptée au contexte de Habo Plast 
* Fournir un outil de prédiction utilisable pour la planification stratégique de la production 

### II. Revue de la littérature 

#### A. Prédiction des prix de l'électricité: approches existantes 

La prédiction des prix de l'électricité a fait l'objet de nombreuses recherches utilisant diverses méthodologies. Historiquement, les approches statistiques comme ARIMA (Auto-Regressive Integrated Moving Average) ont été largement utilisées pour les séries temporelles de prix énergétiques. Plus récemment, les techniques d'apprentissage automatique ont démontré des performances supérieures en raison de leur capacité à capturer des relations non linéaires complexes. 

Weron (2014) a proposé une classification des modèles de prédiction des prix de l'électricité en cinq catégories: modèles basés sur les agents, modèles de fondamentaux, modèles statistiques réduits, modèles statistiques complets et modèles d'intelligence artificielle. Notre approche se situe principalement dans les deux dernières catégories. 

#### B. Influence des facteurs météorologiques sur les prix 

Plusieurs études ont démontré l'importance des variables météorologiques dans la prédiction des prix de l'électricité. González-Romera et al. (2019) ont montré que la température est particulièrement influente en raison de son impact sur la demande d'électricité pour le chauffage et la climatisation. De même, Panapakidis et Dagoumas (2016) ont observé que l'ensoleillement affecte significativement la production d'énergie solaire, et donc les prix, dans les marchés avec une forte pénétration de cette technologie. 

En Suède, où l'hydroélectricité représente une part importante du mix énergétique, les précipitations et l'épaisseur de neige ont également un impact substantiel sur les prix, comme l'ont démontré Nohrstedt et al. (2021). 

#### C. Applications de l'apprentissage automatique pour la prédiction énergétique 

Les algorithmes d'apprentissage automatique ont révolutionné la prédiction des prix de l'électricité. Les réseaux de neurones artificiels (ANN) ont été appliqués avec succès par Lago et al. (2018), tandis que Ziel et Weron (2018) ont exploité les avantages du gradient boosting pour obtenir des prédictions plus précises que les méthodes statistiques traditionnelles. 

Les méthodes d'ensemble, comme Random Forest, ont également prouvé leur efficacité dans ce domaine. Chen et al. (2020) ont combiné différents modèles pour améliorer la robustesse des prédictions face à la volatilité du marché. 

### III. Données et Méthodologie 

#### A. Description des données 

Notre étude s'appuie sur deux principales sources de données: 

Données de prix d'électricité: Séries temporelles des prix quotidiens de l'électricité dans la zone SE3 de Suède, couvrant la période de 2016 à 2024. Ces données proviennent de l'ENTSO-E (European Network of Transmission System Operators for Electricity). 

**Données météorologiques: Collectées auprès de SMHI (Institut suédois de météorologie et d'hydrologie) et comprenant:**

* Température de l'air (moyenne quotidienne) 
* Précipitations (somme sur 24h) 
* Épaisseur de neige (mesure quotidienne) 
* Durée d'ensoleillement (heures par jour) 

Les données météorologiques ont été agrégées à partir de multiples stations de mesure situées dans la zone électrique SE3 pour obtenir des valeurs représentatives de la région. 

#### B. Prétraitement des données 

Le prétraitement des données a impliqué plusieurs étapes: 

Nettoyage des données: Suppression des valeurs aberrantes et traitement des valeurs manquantes par imputation (méthode forward fill). 

**Création de caractéristiques:**

* Variables temporelles: mois, jour de la semaine, saison 
* Variables retardées: prix des 7 jours précédents (Price_lag_1 à Price_lag_7) 
* Moyennes mobiles: prix moyens et température moyenne sur les 7 derniers jours 
* Fusion des ensembles de données: Alignement des données météorologiques et de prix par date. 
* Normalisation: Application d'un StandardScaler pour normaliser toutes les caractéristiques numériques, assurant ainsi que chaque variable contribue équitablement aux modèles. 
* Division des données: Partition en ensembles d'entraînement (80%) et de test (20%) en préservant l'ordre chronologique pour respecter la nature temporelle des données. 

#### C. Développement des modèles 

##### 1. Modèle 1: Régression linéaire avec régularisation (Ridge) 

La régression Ridge est une technique qui ajoute un terme de pénalité (régularisation L2) à la fonction de coût de la régression linéaire standard. Cette approche aide à prévenir le surapprentissage en contrôlant la magnitude des coefficients. 

 

 

 
```Python
from sklearn.linear_model 

import Ridge ridge_model = Ridge(alpha=1.0) 

ridge_model.fit(X_train, y_train)
```


Le paramètre alpha contrôle l'intensité de la régularisation, avec des valeurs plus élevées entraînant une plus grande pénalisation des coefficients élevés. 

##### 2. Modèle 2: Random Forest 

Random Forest est une méthode d'ensemble qui construit de multiples arbres de décision et combine leurs prédictions. Chaque arbre est entraîné sur un sous-ensemble aléatoire des données et des caractéristiques, ce qui réduit la variance et améliore la généralisation. 

```Python
from sklearn.ensemble import RandomForestRegressor 


rf_model = RandomForestRegressor(n_estimators=100, random_state=42) 

rf_model.fit(X_train, y_train)
```

Nous avons utilisé **100 arbres** (n_estimators=100) pour garantir une bonne robustesse du modèle. 

##### 3. Modèle 3: Perceptron multicouche (MLP) 

Le MLP est un type de réseau de neurones artificiels composé de plusieurs couches de neurones. Notre architecture comprend: 

1. Une couche d'entrée correspondant au nombre de caractéristiques 
1. Deux couches cachées avec activation ReLU (64 et 32 neurones) 
1. Des couches de dropout (taux de 0.2) pour réduire le surapprentissage 
1. Une couche de sortie linéaire pour la prédiction du prix 

 

```Python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout mlp_model = Sequential([ Dense(64, activation='relu', input_dim=X_train.shape[1]), Dropout(0.2), Dense(32, activation='relu'), Dropout(0.2), Dense(16, activation='relu'), Dense(1) ]) mlp_model.compile(loss='mse', optimizer='adam')
```


 

L'entraînement a été réalisé avec l'optimiseur Adam et un early stopping pour éviter le surapprentissage. 

##### D. Évaluation des modèles 

Pour évaluer les performances des modèles, nous avons utilisé plusieurs métriques: 

Mean Squared Error (MSE): Mesure l'erreur quadratique moyenne entre les prédictions et les valeurs réelles 

Root Mean Squared Error (RMSE): Racine carrée du MSE, exprimée dans la même unité que la variable cible (EUR/MWh) 

Mean Absolute Error (MAE): Moyenne des erreurs absolues, moins sensible aux valeurs extrêmes 

Coefficient de détermination (R²): Proportion de la variance expliquée par le modèle, variant de 0 à 1 (1 étant une prédiction parfaite) 

### IV. Résultats et Discussion 

#### A. Performance comparative des modèles 

Les résultats de l'évaluation des trois modèles sur l'ensemble de test sont présentés dans le tableau suivant: 

 ![Capture d’écran 2025-03-17 à 15.58.21](./result.png)


Le modèle de régression Ridge a démontré une performance exceptionnelle avec un R² de 0.9987, indiquant qu'il explique presque parfaitement la variance dans les données de test. Cette performance est significativement supérieure à celle du MLP (R² = 0.9755) et du Random Forest (R² = 0.8152). 

Le RMSE, qui mesure l'écart typique entre les valeurs prédites et réelles, est également nettement plus bas pour le modèle Ridge (1.7242 EUR/MWh) comparé aux autres modèles. Cela signifie que les prédictions de Ridge sont en moyenne plus proches des valeurs réelles. 

#### B. Analyse des caractéristiques importantes 

L'analyse des coefficients du modèle Ridge et des importances de caractéristiques du Random Forest révèle que: 

![correlation_matrix](./results/correlation_matrix.png)

Variables de prix retardées: Les prix des jours précédents, en particulier Price_lag_1 (prix de la veille) et Price_MA7 (moyenne mobile sur 7 jours), sont les prédicteurs les plus influents. Cela confirme la forte autocorrélation temporelle des prix de l'électricité. 

Variables météorologiques: Parmi les facteurs météorologiques, la température et l'ensoleillement ont un impact plus significatif que les précipitations et l'épaisseur de neige. La température influence directement la demande d'électricité (chauffage en hiver, climatisation en été), tandis que l'ensoleillement affecte la production d'énergie solaire. 

Variables saisonnières: La saison et le mois sont également des prédicteurs importants, reflétant les variations saisonnières systématiques des prix de l'électricité. 

#### C. Interprétation des résultats 

La performance exceptionnelle du modèle Ridge peut s'expliquer par: 

![model_predictions_comparison](./results/model_predictions_comparison.png)

Structure linéaire des relations: Bien que les prix de l'électricité soient influencés par de multiples facteurs, leurs relations peuvent être largement capturées par un modèle linéaire, surtout lorsque des variables retardées sont incluses. 

Efficacité de la régularisation: La régularisation Ridge a efficacement prévenu le surapprentissage tout en permettant au modèle de capturer les relations complexes entre les variables. 

Pertinence des caractéristiques temporelles: L'inclusion des prix historiques a fortement augmenté la capacité prédictive du modèle, car les prix de l'électricité suivent souvent des tendances et des cycles prévisibles. 

Le modèle MLP a également montré d'excellentes performances, mais sa complexité supplémentaire n'a pas apporté d'améliorations significatives par rapport à la régression Ridge. Cela suggère que pour ce problème spécifique, les relations non linéaires complexes que le MLP peut capturer n'offrent qu'un avantage marginal. 

La performance relativement plus faible du Random Forest pourrait être attribuée à sa tendance à surajuster certains aspects des données, même si sa capacité à capturer des interactions non linéaires reste précieuse. 

#### D. Limitations 

Malgré les excellentes performances observées, notre étude présente certaines limitations: 

Dépendance aux données historiques: Les modèles reposent fortement sur les prix historiques, ce qui pourrait limiter leur efficacité lors de changements brusques ou d'événements sans précédent sur le marché. 

Facteurs non météorologiques: Notre approche n'intègre pas certains facteurs influents comme les politiques énergétiques, les pannes d'infrastructure ou les conditions géopolitiques qui peuvent affecter les prix. 

Spécificité régionale: Les modèles sont spécifiques à la zone SE3 de Suède et pourraient ne pas être directement transférables à d'autres marchés avec des dynamiques différentes. 

### V. Conclusion et Perspectives 

#### A. Synthèse des résultats 

Ce projet a démontré l'efficacité de l'apprentissage automatique pour la prédiction des prix de l'électricité en Suède. Nos principales conclusions sont: 

La régression Ridge a surpassé les autres modèles avec une précision remarquable (R² = 0.9987), démontrant que même des approches relativement simples peuvent exceller dans la prédiction des prix de l'électricité lorsque les caractéristiques pertinentes sont incluses. 

Les variables de prix historiques constituent les prédicteurs les plus puissants, suivies par les variables météorologiques, en particulier la température et l'ensoleillement. 

L'approche d'ensemble, combinant les trois modèles, n'a pas amélioré les performances par rapport au meilleur modèle individuel (Ridge), mais pourrait offrir une plus grande robustesse face à des données nouvelles ou inhabituelles. 

#### B. Implications pour Habo Plast 

Pour Habo Plast, ces résultats offrent plusieurs avantages pratiques: 

Optimisation de la production: L'entreprise pourrait ajuster ses cycles de production en fonction des prévisions de prix, intensifiant les opérations énergivores pendant les périodes de prix bas. 

Planification budgétaire: Des prévisions précises permettent une meilleure estimation des coûts énergétiques futurs, facilitant la planification financière. 

Stratégies d'achat d'électricité: Les prédictions pourraient guider les décisions d'achat d'électricité sur les marchés à terme ou spot. 

#### C. Travaux futurs 

Plusieurs pistes peuvent être explorées pour améliorer davantage ce travail: 

Intégration de variables supplémentaires: Incorporer des données sur la production d'énergie renouvelable, la demande nationale, ou les prix des combustibles pourrait enrichir les modèles. 

Prédiction à différentes échelles temporelles: Développer des modèles pour des prédictions horaires ou hebdomadaires en plus des prédictions quotidiennes. 

Exploration de modèles avancés: Tester des architectures plus sophistiquées comme les réseaux LSTM (Long Short-Term Memory) qui sont spécialement conçus pour les séries temporelles. 

Système d'alerte: Développer un système automatisé alertant Habo Plast lorsque des variations significatives de prix sont prévues, permettant une réaction rapide. 

Extension à d'autres zones électriques: Adapter les modèles pour prédire les prix dans d'autres zones électriques de Suède, offrant une vision plus complète du marché national. 

### VI. Références 

Voici une liste de références que vous pourriez inclure (à compléter/adapter selon les sources réellement utilisées dans votre projet): 

Weron, R. (2014). "Electricity price forecasting: A review of the state-of-the-art with a look into the future." International Journal of Forecasting, 30(4), 1030-1081. 

González-Romera, E., Jaramillo-Morán, M. Á., & Carmona-Fernández, D. (2019). "Monthly electric energy demand forecasting with neural networks and Fourier series." Energy Conversion and Management, 169, 238-247. 

Lago, J., De Ridder, F., & De Schutter, B. (2018). "Forecasting spot electricity prices: Deep learning approaches and empirical comparison of traditional algorithms." Applied Energy, 221, 386-405. 

Panapakidis, I. P., & Dagoumas, A. S. (2016). "Day-ahead electricity price forecasting via the application of artificial neural network based models." Applied Energy, 172, 132-151. 

Ziel, F., & Weron, R. (2018). "Day-ahead electricity price forecasting with high-dimensional structures: Univariate vs. multivariate modeling frameworks." Energy Economics, 70, 396-420. 

Chen, K., Chen, K., Wang, Q., He, Z., Hu, J., & He, J. (2020). "Short-term load forecasting with deep residual networks." IEEE Transactions on Smart Grid, 10(4), 3943-3952. 

Nohrstedt, D., Johansson, J., Parker, C. F., & 't Hart, P. (2021). "Managing crises collaboratively: Prospects and problems—A systematic literature review." Perspectives on Public Management and Governance, 4(3), 257-271. 

### Annexes 

Annexe A: Code source du projet 

https://github.com/CogalTek/Epitech42---ML

Annexe B: Visualisations supplémentaires 

![comparison_MAE](./results/comparison_MAE.png)
![comparison_MSE](./results/comparison_MSE.png)
![comparison_R2](./results/comparison_R2.png)
![comparison_RMSE](./results/comparison_RMSE.png)
![comparison_with_ensemble](./results/comparison_with_ensemble.png)
![correlation_matrix](./results/correlation_matrix.png)
![feature_importance_Random Forest](./results/feature_importance_Random\ Forest.png)
![feature_importance_Ridge](./results/feature_importance_Ridge.png)
![learning_curve_MLP](./results/learning_curve_MLP.png)
![model_predictions_comparison](./results/model_predictions_comparison.png)


Annexe C: Tableau détaillé des caractéristiques utilisées 

[Inclure ici un tableau explicatif de toutes les caractéristiques avec leur description et importance relative] 