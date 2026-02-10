import numpy as np
import pandas as pd
from scipy.signal import detrend

# Charger les données nécessaires
data_path = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\walker_parameters_annual.csv"
data = pd.read_csv(data_path)

# Vérifier les colonnes disponibles
print("Colonnes disponibles dans le fichier :")
print(data.columns.tolist())

# Paramètres
Q0 = 1000  # Exemple de valeur pour Q0
alpha = 0.1  # Exemple de valeur pour alpha
CAPE_max = np.max(data['cape'])  # Valeur maximale de CAPE
N0 = 1  # Exemple de valeur pour N0

# Vérifier que les colonnes nécessaires existent
required_columns = ['Year', 'cape', 'wind_speed', 'wind_direction']
for column in required_columns:
    if column not in data.columns:
        raise ValueError(f"La colonne {column} est manquante dans les données.")

# Calcul de Q_absorbed
data['Q_absorbed'] = Q0 * (1 - np.exp(-alpha * (data['Year'] - data['Year'].min()))) * (data['cape'] / CAPE_max)

# Calcul de lambda_W (nécessite des données de SSTa et kappa_B)
# Si ces colonnes n'existent pas, vous devez les ajouter ou les calculer à partir d'autres données
if 'kappa_B' in data.columns and 'SSTa' in data.columns:
    data['lambda_W'] = data['kappa_B'] * data['SSTa'] * data['Q_absorbed']
else:
    print("Les colonnes 'kappa_B' ou 'SSTa' sont manquantes. Impossible de calculer 'lambda_W'.")

# Calcul de Beta_W (nécessite des données de Delta_W et Delta_SSTa)
# Si ces colonnes n'existent pas, vous devez les ajouter ou les calculer à partir d'autres données
if 'Delta_W' in data.columns and 'Delta_SSTa' in data.columns:
    data['Beta_W'] = data['Delta_W'] / data['Delta_SSTa']
else:
    print("Les colonnes 'Delta_W' ou 'Delta_SSTa' sont manquantes. Impossible de calculer 'Beta_W'.")

# Calcul de N_rising (nécessite des données de Delta_U et W)
# Si ces colonnes n'existent pas, vous devez les ajouter ou les calculer à partir d'autres données
if 'Delta_U' in data.columns and 'W' in data.columns:
    data['N_rising'] = (N0 * data['cape']) / (data['Delta_U'] * data['W'])
else:
    print("Les colonnes 'Delta_U' ou 'W' sont manquantes. Impossible de calculer 'N_rising'.")

# Sauvegarder les résultats dans un fichier CSV
output_path = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data\dynamic_parameters.csv"
data.to_csv(output_path, index=False)

print("\nLes paramètres dynamiques ont été sauvegardés dans 'dynamic_parameters.csv'.")
