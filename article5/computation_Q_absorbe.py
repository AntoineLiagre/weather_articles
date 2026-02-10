import pandas as pd
import numpy as np
import os

# Configuration des chemins de fichiers
output_dir = r"C:\Users\antoi\Desktop\python projects\weather_article456_project\data"
data_file_path = os.path.join(output_dir, "walker_parameters_annual.csv")

# Charger les données nécessaires
data = pd.read_csv(data_file_path)

# Paramètres pour le calcul de Q_absorbed
Q0 = 1000  # Exemple de valeur pour Q0
alpha = 0.1  # Exemple de valeur pour alpha
CAPE_max = np.max(data['cape'])  # Valeur maximale de CAPE

# Calcul de Q_absorbed
data['Q_absorbed'] = Q0 * (1 - np.exp(-alpha * (data['Year'] - data['Year'].min()))) * (data['cape'] / CAPE_max)

# Sauvegarder les résultats dans un fichier CSV
output_path = os.path.join(output_dir, "walker_parameters_annual_with_Q_absorbed.csv")
data.to_csv(output_path, index=False)

print(f"\nLes valeurs de Q_absorbed ont été ajoutées et sauvegardées dans '{output_path}'.")
