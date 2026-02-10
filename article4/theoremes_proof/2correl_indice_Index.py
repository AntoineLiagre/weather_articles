import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
data_path = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\annual_spectral_parameters.csv"
data = pd.read_csv(data_path)

# Extraire les paramètres spectraux et les indices climatiques
lambda_val = data['lambda']
beta = data['beta']
gamma = data['gamma']
kappa_B = data['kappa_B_global']
D_max = data['D_max']
Q_max = data['Q_max']

# Exemple d'indice climatique (à remplacer par vos données réelles)
# Supposons que vous avez une colonne 'ENSO' dans vos données
ENSO = data['ENSO']  # Assurez-vous que cette colonne existe ou utilisez une autre colonne pertinente

# Calculer les corrélations
corr_lambda_ENSO, p_value_lambda_ENSO = pearsonr(lambda_val, ENSO)
corr_beta_ENSO, p_value_beta_ENSO = pearsonr(beta, ENSO)
corr_gamma_ENSO, p_value_gamma_ENSO = pearsonr(gamma, ENSO)
corr_kappa_B_ENSO, p_value_kappa_B_ENSO = pearsonr(kappa_B, ENSO)
corr_D_max_ENSO, p_value_D_max_ENSO = pearsonr(D_max, ENSO)
corr_Q_max_ENSO, p_value_Q_max_ENSO = pearsonr(Q_max, ENSO)

# Afficher les résultats
print(f"Corrélation entre lambda et ENSO: {corr_lambda_ENSO}, p-value: {p_value_lambda_ENSO}")
print(f"Corrélation entre beta et ENSO: {corr_beta_ENSO}, p-value: {p_value_beta_ENSO}")
print(f"Corrélation entre gamma et ENSO: {corr_gamma_ENSO}, p-value: {p_value_gamma_ENSO}")
print(f"Corrélation entre kappa_B et ENSO: {corr_kappa_B_ENSO}, p-value: {p_value_kappa_B_ENSO}")
print(f"Corrélation entre D_max et ENSO: {corr_D_max_ENSO}, p-value: {p_value_D_max_ENSO}")
print(f"Corrélation entre Q_max et ENSO: {corr_Q_max_ENSO}, p-value: {p_value_Q_max_ENSO}")

# Visualiser les relations
plt.figure(figsize=(12, 8))
sns.scatterplot(x=lambda_val, y=ENSO)
plt.title(f"Relation entre lambda et ENSO (Corrélation: {corr_lambda_ENSO:.2f})")
plt.xlabel("lambda")
plt.ylabel("ENSO")
plt.grid(True)
plt.show()
