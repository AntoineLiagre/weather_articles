import pandas as pd
import numpy as np

# Charger les données de n(t)
data_path = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\annual_spectral_parameters.csv"
data = pd.read_csv(data_path)

# Extraire n(t)
n_t = data['n_t']

# Calculer la moyenne mobile sur une fenêtre de 5 ans
window_size = 5
n_t_rolling_mean = n_t.rolling(window=window_size, center=False).mean()

# Ajouter la moyenne mobile au DataFrame
data['n_t_rolling_mean'] = n_t_rolling_mean

# Visualiser les résultats
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data['Year'], data['n_t'], label='n(t)')
plt.plot(data['Year'], data['n_t_rolling_mean'], label=f'n(t) - Moyenne Mobile ({window_size} ans)')
plt.title('Évolution de n(t) avec Moyenne Mobile')
plt.xlabel('Année')
plt.ylabel('n(t)')
plt.legend()
plt.grid(True)
plt.show()
