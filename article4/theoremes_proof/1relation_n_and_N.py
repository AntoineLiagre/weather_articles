import pandas as pd
from scipy.stats import pearsonr

# Charger les données de n(t)
n_t_data = pd.read_csv(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\annual_spectral_parameters.csv")

# Charger les données de stratification
stratification_data = xr.open_dataset(r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\stratification.nc").to_dataframe()

# Fusionner les données
merged_data = pd.merge(n_t_data, stratification_data, on='Year')

# Calculer les corrélations
corr, p_value = pearsonr(merged_data['n_t'], merged_data['dN'])
print(f"Corrélation de Pearson entre n(t) et ΔN: {corr}, p-value: {p_value}")
