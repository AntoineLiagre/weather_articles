import pandas as pd
import numpy as np

# Charger les données
data_path = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\annual_spectral_parameters.csv"
data = pd.read_csv(data_path)

# Extraire les variables nécessaires
n_t = data['n_t']
lambda_val = data['lambda']
beta = data['beta']
gamma = data['gamma']
kappa_B = data['kappa_B_global']
D_max = data['D_max']
Q_max = data['Q_max']

# Créer un DataFrame avec les variables indépendantes et dépendantes
df = pd.DataFrame({
    'n_t': n_t,
    'lambda': lambda_val,
    'beta': beta,
    'gamma': gamma,
    'kappa_B': kappa_B,
    'D_max': D_max,
    'Q_max': Q_max
})

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()
